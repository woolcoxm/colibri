/* Production-scale A/B for the ragged attention kernel.
 *
 * The shipped test_ragged_attention.cu checks ragged vs batch at toy dimensions
 * (K=3, T=3, H=2). That passed (ragged_relative_rms=0) on RTX 5070 Ti. This test
 * repeats the comparison at GLM-5.2's real MLA dimensions (K=512, Q=192, R=64,
 * V=256, H=64) and with realistic ragged lengths (independent KV sequences of
 * differing prefix lengths). Both arms run on the SAME GPU tensors (ragged CUDA
 * path vs the per-sequence batch CUDA path), so this isolates the ragged kernel
 * with no CPU / model / routing confound.
 *
 * Gate: ragged_relative_rms < 1e-6 (bit-equivalence). */
#include "../backend_cuda.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

int main(){
    int dev=0;if(!coli_cuda_init(&dev,1))return 77;
    /* Real GLM-5.2 MLA dims. */
    constexpr int H=64,Q=192,R=64,V=256,K=512,D=H*V,O=8,T=40;
    constexpr int S=4;                       /* 4 independent KV rows */
    int n[S]={8,17,31,40};                   /* ragged: different valid prefix each row */

    std::vector<float> w((size_t)H*(Q+V)*K), p((size_t)O*D), q((size_t)S*H*(Q+R));
    /* small deterministic patterned weights (sign-symmetric, magnitude-controlled) */
    unsigned st=12345u; auto rng=[&](){ st^=st<<13; st^=st>>7; st^=st<<17; return (st/4294967296.0)-0.5; };
    for(size_t i=0;i<w.size();i++) w[i]=rng()*0.10f;
    for(size_t i=0;i<p.size();i++) p[i]=rng()*0.12f;
    for(size_t i=0;i<q.size();i++) q[i]=rng()*0.08f;

    ColiCudaTensor *tw=nullptr,*tp=nullptr;
    if(!coli_cuda_tensor_upload(&tw,w.data(),nullptr,0,K,H*(Q+V),dev)||
       !coli_cuda_tensor_upload(&tp,p.data(),nullptr,0,D,O,dev)){ std::printf("upload fail\n"); return 1; }

    std::vector<std::vector<float>> l(S),r(S);
    const float *lp[S],*rp[S];
    for(int s=0;s<S;s++){
        l[s].resize((size_t)n[s]*K); r[s].resize((size_t)n[s]*R);
        for(size_t i=0;i<l[s].size();i++) l[s][i]=rng()*0.09f;
        for(size_t i=0;i<r[s].size();i++) r[s][i]=rng()*0.07f;
        lp[s]=l[s].data(); rp[s]=r[s].data();
    }

    /* Arm A: ragged CUDA kernel, all S rows in one launch. */
    std::vector<float> got((size_t)S*O);
    if(!coli_cuda_attention_project_ragged(tw,tp,got.data(),q.data(),lp,rp,n,S,H,Q,R,V,K,T,.02f)){
        std::printf("ragged launch fail\n"); return 2; }

    /* Arm B: per-sequence batch CUDA path (one row at a time) — same GPU tensors. */
    std::vector<float> ref((size_t)S*O);
    for(int s=0;s<S;s++)
        if(!coli_cuda_attention_project_batch(tw,tp,ref.data()+s*O,
            q.data()+(size_t)s*H*(Q+R),lp[s],rp[s],1,H,Q,R,V,K,n[s],.02f)){
            std::printf("batch launch fail row %d\n",s); return 3; }

    /* Arm C: ragged again but S=1 per row — isolates multi-row interaction. If C
     * matches B but A (S=4) doesn't, the bug is in multi-row batching. */
    std::vector<float> sing((size_t)S*O);
    for(int s=0;s<S;s++){
        int one=1;
        if(!coli_cuda_attention_project_ragged(tw,tp,sing.data()+s*O,
            q.data()+(size_t)s*H*(Q+R),lp+s,rp+s,&n[s],1,H,Q,R,V,K,n[s],.02f)){
            std::printf("ragged S=1 launch fail row %d\n",s); return 5; }
    }

    auto rms_of=[&](const std::vector<float>& a,const std::vector<float>& b)->double{
        double e=0,z=0,absmax=0;
        for(int i=0;i<S*O;i++){double d=(double)a[i]-b[i];e+=d*d;z+=(double)b[i]*b[i];
            double ad=d<0?-d:d;if(ad>absmax)absmax=ad;}
        std::printf("   rms=%.9g max_abs=%.9g ref_rms=%.9g\n",
            std::sqrt(e/(z+1e-30)),absmax,std::sqrt(z/(S*O)));
        return std::sqrt(e/(z+1e-30));
    };
    std::printf("real-scale (K=%d Q=%d R=%d V=%d H=%d S=%d lengths={%d,%d,%d,%d})\n",
        K,Q,R,V,H,S,n[0],n[1],n[2],n[3]);
    std::printf("A vs B  (ragged S=4  vs  batch S=1 each):\n"); double ab=rms_of(got,ref);
    std::printf("C vs B  (ragged S=1  vs  batch S=1 each):\n"); double cb=rms_of(sing,ref);
    std::printf("A vs C  (ragged S=4  vs  ragged S=1 each):\n"); double ac=rms_of(got,sing);
    int fail = (ab>=1e-6);
    std::printf(fail? "FAIL\n" : "PASS\n");
    coli_cuda_tensor_free(tw); coli_cuda_tensor_free(tp); coli_cuda_shutdown();
    return fail?4:0;
}
