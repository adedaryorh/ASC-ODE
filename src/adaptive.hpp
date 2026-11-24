#pragma once
#include "explicitRK.hpp"
#include <algorithm>
#include <limits>
namespace ASC_ode {
// trial writes the higher-order estimate and the componentwise difference.
class EmbeddedTimeStepper {
public:
  virtual ~EmbeddedTimeStepper()=default;
  virtual int order() const=0;
  virtual void trial(double h,VectorView<> y,VectorView<> difference)=0;
};
class StepperPair : public EmbeddedTimeStepper {
  TimeStepper &low,&high;
  int lowOrder;
public:
  StepperPair(TimeStepper& low, TimeStepper& high,int p) : low(low),high(high),lowOrder(p) {}
  int order() const override {return lowOrder;}
  void trial(double h,VectorView<> y,VectorView<> difference) override {
    difference=y;
    low.doStep(h,difference); high.doStep(h,y);
    difference=y-difference;
  }
};
// Any explicit tableau with two weight rows can share the same stage slopes.
class EmbeddedExplicitRK : public ExplicitRungeKutta, public EmbeddedTimeStepper {
  Vector<> lowerWeights;
  int lowerOrder;
public:
  EmbeddedExplicitRK(std::shared_ptr<NonlinearFunction> rhs,const Matrix<>& a,
      const Vector<>& high,const Vector<>& low,const Vector<>& c,int p)
    : ExplicitRungeKutta(rhs,a,high,c),lowerWeights(low),lowerOrder(p) {
    if(low.size()!=high.size() || p<1) throw std::invalid_argument("Invalid embedded weights or order");
  }
  int order() const override {return lowerOrder;}
  void trial(double h,VectorView<> y,VectorView<> difference) override {
    stages(h,y);
    difference=0.;
    for(size_t i=0;i<weights.size();i++) difference+=(h*(weights(i)-lowerWeights(i)))*slopes.row(i);
    advance(h,y);
  }
};
class BogackiShampine : public EmbeddedExplicitRK {
public:
  explicit BogackiShampine(std::shared_ptr<NonlinearFunction> rhs)
    : EmbeddedExplicitRK(rhs,{{0,0,0,0},{.5,0,0,0},{0,.75,0,0},{2./9,1./3,4./9,0}},
      {2./9,1./3,4./9,0},{7./24,.25,1./3,.125},{0,.5,.75,1},2) {}
};
struct StepCounts {size_t accepted=0,rejected=0;};
using Observer=std::function<void(double,double,VectorView<>)>;
inline StepCounts SolveODE_Adaptive(EmbeddedTimeStepper& method,double end,double h,
    double atol,double rtol,VectorView<> y,Observer observe={}) {
  if(atol<=0 || rtol<0 || h<=0 || end<0) throw std::invalid_argument("Invalid step or tolerance");
  Vector<> trial(y.size()),difference(y.size());
  StepCounts counts;
  double t=0;
  while(t<end) {
    h=std::min(h,end-t);
    if(t+h==t) throw std::runtime_error("Time-step underflow");
    trial=y;
    double error=0;
    try {
      method.trial(h,trial,difference);
      for(size_t j=0;j<y.size();j++) {
        double scale=atol+rtol*std::max(std::abs(y(j)),std::abs(trial(j)));
        double ratio=difference(j)/scale;
        error+=ratio*ratio;
      }
      error=std::sqrt(error/y.size());
    } catch(const std::domain_error&) {error=std::numeric_limits<double>::infinity();}
      catch(const std::runtime_error&) {error=std::numeric_limits<double>::infinity();}
    if(!std::isfinite(error)) {
      ++counts.rejected; h*=.5; continue;
    }
    if(error<=1) {
      y=trial; t+=h; ++counts.accepted;
      if(observe) observe(t,h,y);
    } else ++counts.rejected;
    double factor=error==0 ? 5. : .9*std::pow(error,-1./(method.order()+1));
    h*=std::clamp(factor,.2,5.);
  }
  return counts;
}
}
