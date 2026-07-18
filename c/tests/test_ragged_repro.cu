/* Minimal repro: does running S=1 batch calls BEFORE an S=4 ragged call
 * corrupt the S=4 result? (This mirrors what the real model + our A/B do:
 * many calls reuse the DeviceContext scratch.) */
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
    std::vector<float> qflat((size_t)S*H*(Q+R));
    for(int s=0;s<S;s++)for(size_t i=0;i<(size_t)H*(Q+R);i++)qflat[(size_t)s*H*(Q+R)+i]=qv[s][i];
    const float* lp[4]={lv[0].data(),lv[1].data(),lv[2].data(),lv[3].data()};
    const float* rp[4]={rv[0].data(),rv[1].data(),rv[2].data(),rv[3].data()};
    int n[4]={T,T,T,T};

    /* Run 1: ragged S=4 FIRST (cold dc). */
    std::vector<float> cold((size_t)S*O);
    coli_cuda_attention_project_ragged(tw,tp,cold.data(),qflat.data(),lp,rp,n,S,H,Q,R,V,K,T,.02f);

    /* Pollute dc with S=1 batch calls (like building a reference). */
    std::vector<float> dummy(O);
    for(int s=0;s<S;s++)coli_cuda_attention_project_batch(tw,tp,dummy.data(),qv[s].data(),lv[s].data(),rv[s].data(),1,H,Q,R,V,K,T,.02f);

    /* Run 2: ragged S=4 AGAIN, same inputs, same tensors. Must be bit-identical to Run 1. */
    std::vector<float> warm((size_t)S*O);
    coli_cuda_attention_project_ragged(tw,tp,warm.data(),qflat.data(),lp,rp,n,S,H,Q,R,V,K,T,.02f);

    double e=0,z=0,mx=0;
    for(size_t i=0;i<(size_t)S*O;i++){double d=(double)cold[i]-warm[i];e+=d*d;z+=(double)cold[i]*cold[i];
        double ad=d<0?-d:d;if(ad>mx)mx=ad;}
    std::printf("cold-vs-warm (same inputs, dc polluted between): rms=%.9g max_abs=%.9g\n",
        std::sqrt(e/(z+1e-30)),mx);
    /* also vs the S=1 batch reference */
    std::vector<float> ref((size_t)S*O);
    for(int s=0;s<S;s++)coli_cuda_attention_project_batch(tw,tp,ref.data()+s*O,qv[s].data(),lv[s].data(),rv[s].data(),1,H,Q,R,V,K,T,.02f);
    double e2=0,z2=0;for(size_t i=0;i<(size_t)S*O;i++){double d=(double)warm[i]-ref[i];e2+=d*d;z2+=(double)ref[i]*ref[i];}
    std::printf("warm-vs-batchref: rms=%.9g\n",std::sqrt(e2/(z2+1e-30)));
    std::printf("cold-vs-batchref: rms=%.9g\n",
        [&]{double ee=0,zz=0;for(size_t i=0;i<(size_t)S*O;i++){double d=(double)cold[i]-ref[i];ee+=d*d;zz+=(double)ref[i]*ref[i];}return std::sqrt(ee/(zz+1e-30));}());
    coli_cuda_tensor_free(tw);coli_cuda_tensor_free(tp);coli_cuda_shutdown();
    return 0;
}
