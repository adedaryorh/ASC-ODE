#pragma once
#include "implicitRK.hpp"
namespace ASC_ode {
class ExplicitRungeKutta : public TimeStepper {
protected:
  Matrix<> coefficients, slopes;
  Vector<> weights, stage;
  void stages(double h,VectorView<> y) {
    for(size_t i=0;i<weights.size();i++) {
      stage=y;
      for(size_t j=0;j<i;j++) stage+= (h*coefficients(i,j))*slopes.row(j);
      m_rhs->evaluate(stage,slopes.row(i));
    }
  }
  void advance(double h,VectorView<> y) const {
    for(size_t i=0;i<weights.size();i++) y+=(h*weights(i))*slopes.row(i);
  }
public:
  ExplicitRungeKutta(std::shared_ptr<NonlinearFunction> rhs,const Matrix<>& a,
                     const Vector<>& b,const Vector<>& c)
    : TimeStepper(rhs),coefficients(a),slopes(b.size(),rhs->dimF()),weights(b),stage(rhs->dimX()) {
    if(!b.size() || a.rows()!=b.size() || a.cols()!=b.size() || c.size()!=b.size())
      throw std::invalid_argument("Inconsistent tableau dimensions");
    for(size_t i=0;i<b.size();i++)
      for(size_t j=i;j<b.size();j++)
        if(a(i,j)!=0) throw std::invalid_argument("Expected an explicit tableau");
  }
  void doStep(double h,VectorView<> y) override { stages(h,y); advance(h,y); }
};
inline ExplicitRungeKutta classicalRK4(std::shared_ptr<NonlinearFunction> rhs) {
  return {rhs,{{0,0,0,0},{.5,0,0,0},{0,.5,0,0},{0,0,1,0}},
    {1./6,1./3,1./3,1./6},{0,.5,.5,1}};
}
}
