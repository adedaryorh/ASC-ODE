#ifndef MASS_SPRING_HPP
#define MASS_SPRING_HPP

#include <nonlinfunc.hpp>
#include <timestepper.hpp>

using namespace ASC_ode;

#include <vector.hpp>
using namespace nanoblas;


template <int D>
class Mass
{
public:
  double mass;
  Vec<D> pos;
  Vec<D> vel = 0.0;
  Vec<D> acc = 0.0;
};


template <int D>
class Fix
{
public:
  Vec<D> pos;
};


// refers to a fix or a mass, by its type and index
class Connector
{
public:
  enum CONTYPE { FIX=1, MASS=2 };
  CONTYPE type;
  size_t nr;
};

inline std::ostream & operator<< (std::ostream & ost, const Connector & con)
{
  ost << "type = " << int(con.type) << ", nr = " << con.nr;
  return ost;
}

class Spring
{
public:
  double length;  
  double stiffness;
  std::array<Connector,2> connectors;
};

struct DistanceConstraint { double length; std::array<Connector,2> connectors; };

template <int D>
class MassSpringSystem
{
  std::vector<Fix<D>> m_fixes;
  std::vector<Mass<D>> m_masses;
  std::vector<Spring> m_springs;
  std::vector<DistanceConstraint> m_constraints;
  Vec<D> m_gravity=0.0;
public:
  void setGravity (Vec<D> gravity) { m_gravity = gravity; }
  Vec<D> getGravity() const { return m_gravity; }

  Connector addFix (Fix<D> p)
  {
    m_fixes.push_back(p);
    return { Connector::FIX, m_fixes.size()-1 };
  }

  Connector addMass (Mass<D> m)
  {
    m_masses.push_back (m);
    return { Connector::MASS, m_masses.size()-1 };
  }
  
  size_t addSpring (Spring s) 
  {
    m_springs.push_back (s); 
    return m_springs.size()-1;
  }

  size_t addConstraint(DistanceConstraint c) {
    m_constraints.push_back(c);
    return m_constraints.size()-1;
  }
  const auto& constraints() const { return m_constraints; }
  auto & fixes() { return m_fixes; } 
  auto & masses() { return m_masses; } 
  auto & springs() { return m_springs; }

  void getState (VectorView<> values, VectorView<> dvalues, VectorView<> ddvalues)
  {
    auto valmat = values.range(0,D*m_masses.size()).asMatrix(m_masses.size(), D);
    auto dvalmat = dvalues.range(0,D*m_masses.size()).asMatrix(m_masses.size(), D);
    auto ddvalmat = ddvalues.range(0,D*m_masses.size()).asMatrix(m_masses.size(), D);

    for (size_t i = 0; i < m_masses.size(); i++)
      {
        valmat.row(i) = m_masses[i].pos;
        dvalmat.row(i) = m_masses[i].vel;
        ddvalmat.row(i) = m_masses[i].acc;
      }
  }

  void setState (VectorView<> values, VectorView<> dvalues, VectorView<> ddvalues)
  {
    auto valmat = values.range(0,D*m_masses.size()).asMatrix(m_masses.size(), D);
    auto dvalmat = dvalues.range(0,D*m_masses.size()).asMatrix(m_masses.size(), D);
    auto ddvalmat = ddvalues.range(0,D*m_masses.size()).asMatrix(m_masses.size(), D);

    for (size_t i = 0; i < m_masses.size(); i++)
      {
        m_masses[i].pos = valmat.row(i);
        m_masses[i].vel = dvalmat.row(i);
        m_masses[i].acc = ddvalmat.row(i);
      }
  }
};

template <int D>
std::ostream & operator<< (std::ostream & ost, MassSpringSystem<D> & mss)
{
  ost << "fixes:" << std::endl;
  for (auto f : mss.fixes())
    ost << f.pos << std::endl;

  ost << "masses: " << std::endl;
  for (auto m : mss.masses())
    ost << "m = " << m.mass << ", pos = " << m.pos << std::endl;

  ost << "springs: " << std::endl;
  for (auto sp : mss.springs())
    ost << "length = " << sp.length << ", stiffness = " << sp.stiffness
        << ", C1 = " << sp.connectors[0] << ", C2 = " << sp.connectors[1] << std::endl;
  return ost;
}


// right-hand side for the Newmark solvers: the accelerations M^{-1} F(x) of all masses
template <int D>
class MSS_Function : public NonlinearFunction
{
  MassSpringSystem<D> & mss;
  size_t physical() const { return D*mss.masses().size(); }
  Vec<D> point(Connector c,VectorView<double> x) const {
    if(c.type==Connector::FIX) return mss.fixes()[c.nr].pos;
    return Vec<D>(x.range(D*c.nr,D*(c.nr+1)));
  }
public:
  explicit MSS_Function(MassSpringSystem<D>& system) : mss(system) {}
  size_t dimX() const override { return physical()+mss.constraints().size(); }
  size_t dimF() const override { return dimX(); }
  void evaluate(VectorView<double> x,VectorView<double> f) const override {
    f=0.;
    for(size_t i=0;i<mss.masses().size();i++)
      f.range(D*i,D*(i+1))=mss.getGravity();
    auto apply=[&](auto ends, const Vec<D>& force) {
      for(int e=0;e<2;e++) {
        auto c=ends[e];
        if(c.type==Connector::MASS)
          f.range(D*c.nr,D*(c.nr+1))+=((e==0?1.:-1.)/mss.masses()[c.nr].mass)*force;
      }
    };
    for(auto& sp:mss.springs()) {
      auto ends=sp.connectors;
      Vec<D> delta=point(ends[1],x)-point(ends[0],x);
      double length=norm(delta);
      if(length==0) throw std::domain_error("Spring endpoints coincide");
      apply(ends,Vec<D>((sp.stiffness*(1-sp.length/length))*delta));
    }
    for(size_t i=0;i<mss.constraints().size();i++) {
      auto c=mss.constraints()[i]; auto ends=c.connectors;
      Vec<D> delta=point(ends[1],x)-point(ends[0],x);
      f(physical()+i)=(norm(delta)*norm(delta))-c.length*c.length;
      apply(ends,Vec<D>((-2*x(physical()+i))*delta));
    }
  }
  void evaluateDeriv(VectorView<double> x,MatrixView<double> df) const override {
    df=0.;
    auto blocks=[&](auto ends,auto stiffness) {
      for(int e=0;e<2;e++) for(int b=0;b<2;b++) {
        auto row=ends[e],col=ends[b];
        if(row.type!=Connector::MASS || col.type!=Connector::MASS) continue;
        double scale=(e==b?-1.:1.)/mss.masses()[row.nr].mass;
        for(int i=0;i<D;i++) for(int j=0;j<D;j++)
          df(D*row.nr+i,D*col.nr+j)+=scale*stiffness(i,j);
      }
    };
    for(auto& spring:mss.springs()) {
      auto ends=spring.connectors;
      Vec<D> delta=point(ends[1],x)-point(ends[0],x);
      double r=norm(delta);
      if(r==0) throw std::domain_error("Spring endpoints coincide");
      blocks(ends,[&](int i,int j){
        return spring.stiffness*((1-spring.length/r)*(i==j)+spring.length*delta(i)*delta(j)/(r*r*r));
      });
    }
    for(size_t n=0;n<mss.constraints().size();n++) {
      auto ends=mss.constraints()[n].connectors;
      size_t row=physical()+n;
      Vec<D> delta=point(ends[1],x)-point(ends[0],x);
      blocks(ends,[&](int i,int j){return -2*x(row)*(i==j);});
      for(int e=0;e<2;e++) if(ends[e].type==Connector::MASS) {
        auto m=ends[e].nr;
        for(int i=0;i<D;i++) {
          double g=(e==0?-2.:2.)*delta(i);
          df(row,D*m+i)+=g; df(D*m+i,row)+=g/mss.masses()[m].mass;
        }
      }
    }
  }
};
#endif
