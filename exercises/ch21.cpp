#include "../mechsystem/mass_spring.hpp"
#include "../mechsystem/verlet.hpp"
#include "functions.hpp"
#include <fstream>
#include <iomanip>
using namespace ASC_ode;

// Initialize multipliers from c'' = grad(c) a + 2 |relative velocity|^2 = 0.
template<int D>
void constrainedRun(MassSpringSystem<D>& system,const std::string& name,
                    double rho,double end,int steps) {
  auto f=std::make_shared<MSS_Function<D>>(system);
  const size_t n=D*system.masses().size(), count=system.constraints().size(), dim=f->dimX();
  Vector<> x(dim),v(dim),a(dim),force(dim); x=0.;v=0.;a=0.;
  system.getState(x,v,a);
  Matrix<> j(dim,dim); f->evaluateDeriv(x,j); f->evaluate(x,force);
  if(count) {
    Matrix<> normal(count,count); Vector<> residual(count);
    for(size_t i=0;i<count;i++) {
      auto ends=system.constraints()[i].connectors;
      Vec<D> relative=0.;
      for(int e=0;e<2;e++) if(ends[e].type==Connector::MASS)
        for(int k=0;k<D;k++) relative(k)+=(e?1.:-1.)*system.masses()[ends[e].nr].vel(k);
      residual(i)=-2*std::pow(norm(relative),2);
      for(size_t k=0;k<n;k++) residual(i)-=j(n+i,k)*force(k);
      for(size_t b=0;b<count;b++) {
        normal(i,b)=0;
        for(size_t k=0;k<n;k++) normal(i,b)+=j(n+i,k)*j(k,n+b);
      }
    }
    calcInverse(normal); x.range(n,dim)=normal*residual;
  }
  f->evaluate(x,force); a.range(0,n)=force.range(0,n);
  // Compare the complete analytical Jacobian (including the multiplier rows)
  // with a central difference at a displaced state.
  Vector<> probe=x,plus(dim),minus(dim);
  for(size_t i=0;i<n;i++) probe(i)+=.01*std::sin(i+1);
  f->evaluateDeriv(probe,j); double worst=0;
  for(size_t col=0;col<dim;col++) {
    const double old=probe(col),eps=1e-6;
    probe(col)=old+eps; f->evaluate(probe,plus);
    probe(col)=old-eps; f->evaluate(probe,minus); probe(col)=old;
    for(size_t row=0;row<dim;row++)
      worst=std::max(worst,std::abs(j(row,col)-(plus(row)-minus(row))/(2*eps)));
  }
  if(worst>1e-5) throw std::runtime_error("Mechanical Jacobian mismatch");
  auto mass=std::make_shared<Projector>(dim,0,n);
  std::ofstream out(name+".csv");out<<"t";
  for(size_t i=0;i<n;i++) out<<",x"<<i;
  for(size_t i=0;i<count;i++) out<<",c"<<i<<",lambda"<<i;
  out<<'\n'<<std::setprecision(15);
  double maxDrift=0;
  auto record=[&](double t,VectorView<> state) {
    out<<t; f->evaluate(state,force);
    for(size_t i=0;i<n;i++) out<<','<<state(i);
    for(size_t i=n;i<dim;i++) {
      out<<','<<force(i)<<','<<state(i);
      maxDrift=std::max(maxDrift,std::abs(force(i)));
    }
    out<<'\n';
  };
  record(0,x); SolveODE_Alpha(end,steps,rho,x,v,a,f,mass,record);
  std::cout<<name<<" max constraint "<<maxDrift<<" Jacobian mismatch "<<worst<<'\n';
}
void structures() {
  for(double rho : {1.,.8}) {
    MassSpringSystem<2> m; m.setGravity({0,-1});
    auto anchor=m.addFix({{0,0}});
    auto bob=m.addMass({1,{.6,-.8}});
    m.addConstraint({1,{anchor,bob}});
    constrainedRun(m,rho==1?"dae_1":"dae_08",rho,10,1000);
  }
  for(int links : {2,4}) {
    MassSpringSystem<2> m; m.setGravity({0,-1});
    auto last=m.addFix({{0,0}});
    for(int i=1;i<=links;i++) {
      auto bob=m.addMass({1,{.6*i,-.8*i}});
      m.addConstraint({1,{last,bob}}); last=bob;
    }
    constrainedRun(m,"chain"+std::to_string(links),.8,4,800);
  }
  MassSpringSystem<2> crane; auto left=crane.addFix({{0,0}});
  auto right=crane.addFix({{1,0}}); auto apex=crane.addMass({1,{.5,1}});
  auto tip=crane.addMass({1,{1.5,1},{0,.2}});
  for(auto base:{left,right}) crane.addSpring({std::sqrt(1.25),100,{base,apex}});
  crane.addSpring({1,100,{apex,tip}}); crane.addSpring({std::sqrt(1.25),100,{right,tip}});
  constrainedRun(crane,"crane",.8,6,600);
  MassSpringSystem<3> top; top.setGravity({0,0,-1}); auto pivot=top.addFix({{0,0,0}});
  std::array<Connector,3> ring;
  for(int i=0;i<3;i++) {
    const double phi=2*std::numbers::pi*i/3;
    // Tilt the symmetry axis by 0.2 radians. Rotate positions and velocities together.
    double X=std::cos(phi),Y=std::sin(phi),c=std::cos(.2),s=std::sin(.2);
    ring[i]=top.addMass({1,{c*X+s,Y,c-s*X},{-8*c*Y,8*X,8*s*Y}});
    top.addConstraint({std::sqrt(2.),{pivot,ring[i]}});
  }
  for(int i=0;i<3;i++) top.addConstraint({std::sqrt(3.),{ring[i],ring[(i+1)%3]}});
  constrainedRun(top,"top",.8,1,500);
}
void energy() {
  auto rhs=std::make_shared<PendulumAD>();
  ExplicitEuler ee(rhs); ImplicitEuler ie(rhs); CrankNicolson cn(rhs);
  const double angle=.5;
  const double period=4*std::comp_ellint_1(std::sin(angle/2));
  int n=100000; double h=1000*period/n;
  std::ofstream out("energy.csv");out<<"t,explicit,implicit,cn,verlet\n"<<std::setprecision(16);
  Vector<> e={angle,0},i=e,c=e;
  double q=angle,v=0;
  auto E=[](double q,double v){return .5*v*v+1-std::cos(q);};
  for(int k=0;k<=n;k++) {
    if(k%10==0) out<<k*h<<','<<E(e(0),e(1))<<','<<E(i(0),i(1))<<','<<E(c(0),c(1))<<','<<E(q,v)<<'\n';
    if(k==n) break;
    ee.doStep(h,e); ie.doStep(h,i); cn.doStep(h,c);
    v-=.5*h*std::sin(q); q+=h*v; v-=.5*h*std::sin(q);
  }
}
void stability() {
  constexpr int n=4;
  MassSpringSystem<1> system; auto left=system.addFix({{0}});
  for(int i=1;i<=n;i++) {
    auto next=system.addMass({1,{double(i)}});
    system.addSpring({1,1,{left,next}}); left=next;
  }
  auto right=system.addFix({{n+1.}}); system.addSpring({1,1,{left,right}});
  const double omega=2*std::sin(n*std::numbers::pi/(2*(n+1)));
  const double critical=2/omega;
  for(double factor:{.99,1.01}) {
    Vector<> x(n),v(n);v=0.;
    for(int j=0;j<n;j++) x(j)=j+1+1e-8*std::sin((j+1)*n*std::numbers::pi/(n+1));
    double amp=0; const int steps=40;
    SolveODE_Verlet(steps*factor*critical,steps,x,v,std::make_shared<MSS_Function<1>>(system),
      [&](double,VectorView<> state){for(int j=0;j<n;j++)amp=std::max(amp,std::abs(state(j)-j-1));});
    std::cout<<"Verlet critical "<<critical<<" h "<<critical*factor<<" max displacement "<<amp<<'\n';
  }
}
int main(){structures();energy();stability();}
