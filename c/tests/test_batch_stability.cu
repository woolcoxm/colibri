/* Control: is the BATCH path itself stable across identical runs? If it also
 * intermittently diverges, then the non-determinism is a harness/driver issue
 * and the ragged conclusion is invalid. Compare batch S=1 run N vs batch S=1
 * run 1, across many runs. */
#include "../backend_cuda.h"
#include <cmath>
#include <cstdio>
#include <vector>
static unsigned st=9998877u;
static float rng(){ st^=st<<13; st^=st>>7; st^=st<<17; return (st/4294967296.0f)-0.5f; }
int main(){
    int dev=0; if(!coli_cuda_init(&dev,1)) return 77;
    const int H=64,Q=192,R=64,V=256,K=512,D=H*V,O=8,T=40,S=4;
    std::vector<float> w((size_t)H*(Q+V)*K),p((size_t)O*D);
    for(auto&x:w)x=rng()*0.10f; for(auto&x:p)x=rng()*0.12f;
    ColiCudaTensor *tw=nullptr,*tp=nullptr;
    if(!coli_cuda_tensor_upload(&tw,w.data(),nullptr,0,K,H*(Q+V),dev)||
       !coli_cuda_tensor_upload(&tp,p.data(),nullptr,0,D,O,dev))return 1;
    std::vector<std::vector<float>> qv(S),lv(S),rv(S);
    for(int s=0;s<S;s++){qv[s].resize(H*(Q+R));lv[s].resize(T*K);rv[s].resize(T*R);
        for(auto&x:qv[s])x=rng()*0.08f;for(auto&x:lv[s])x=rng()*0.09f;for(auto&x:rv[s])x=rng()*0.07f;}
    /* baseline batch run */
    std::vector<float> base((size_t)S*O);
    for(int s=0;s<S;s++)coli_cuda_attention_project_batch(tw,tp,base.data()+s*O,qv[s].data(),lv[s].data(),rv[s].data(),1,H,Q,R,V,K,T,.02f);
    /* compare 8 more identical batch runs to baseline */
    int fails=0;
    for(int it=0; it<8; it++){
        std::vector<float> cur((size_t)S*O);
        for(int s=0;s<S;s++)coli_cuda_attention_project_batch(tw,tp,cur.data()+s*O,qv[s].data(),lv[s].data(),rv[s].data(),1,H,Q,R,V,K,T,.02f);
        double e=0,z=0;for(size_t i=0;i<(size_t)S*O;i++){double d=(double)cur[i]-base[i];e+=d*d;z+=(double)base[i]*base[i];}
        double rms=std::sqrt(e/(z+1e-30));
        std::printf("batch iter %d vs baseline: rms=%.9g %s\n",it,rms,rms<1e-6?"ok":"<<DIVERGE");
        if(rms>=1e-6)fails++;
    }
    std::printf("batch path: %d/8 diverged\n",fails);
    coli_cuda_tensor_free(tw);coli_cuda_tensor_free(tp);coli_cuda_shutdown();
    return 0;
}
