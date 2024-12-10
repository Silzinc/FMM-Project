#include "utils.hpp"
#include <boost/qvm.hpp>
#include <fmt/base.h>
#include <fmt/format.h>

namespace qvm = boost::qvm;

namespace fmm
{
typedef qvm::vec<double, 3> Vec3;
typedef qvm::mat<double, 3, 3> Mat3x3;

namespace plummer
{
double
phi(const fmm::Vec3& xi)
{
  const double xin = qvm::mag(xi);
  return xin / std::sqrt(1.0 + xin * xin);
}

Vec3
grad_phi(const Vec3& xi)
{
  const double xin2 = qvm::mag_sqr(xi);
  return -xi * xin2 / std::pow(1.0 + xin2, 1.5);
}

Mat3x3
hess_phi(const Vec3& xi)
{
  using qvm::X, qvm::Y, qvm::Z;

  const Mat3x3 outer = { X(xi) * X(xi), X(xi) * Y(xi), X(xi) * Z(xi),
                         Y(xi) * X(xi), Y(xi) * Y(xi), Y(xi) * Z(xi),
                         Z(xi) * X(xi), Z(xi) * Y(xi), Z(xi) * Z(xi) };

  const Mat3x3 identity = { 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0 };

  const double xin2 = qvm::mag_sqr(xi);
  const double xin = std::sqrt(xin2);

  return xin * xin * xin *
         (3 * outer / std::pow(1.0 + xin2, 2.5) -
          identity / std::pow(1.0 + xin2, 1.5));
}
}
}
