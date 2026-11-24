#include "functions.hpp"
#include <iomanip>
using namespace ASC_ode;
int main() {
  std::cout<<"x,n,P,derivative\n"<<std::setprecision(16);
  for(int i=0;i<=200;i++) {
    double x=-1+0.01*i;
    for(int n=0;n<=5;n++) {
      auto p=legendre(n,AutoDiff<1>(Variable<0>(x)));
      std::cout<<x<<','<<n<<','<<p.value()<<','<<p.deriv()[0]<<'\n';
    }
  }
}
