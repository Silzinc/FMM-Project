#pragma once
#include <boost/qvm.hpp>
#include <fmt/base.h>
#include <fmt/format.h>

typedef boost::qvm::vec<double, 3> Vec3;
typedef boost::qvm::mat<double, 3, 3> Mat3x3;

template<>
struct fmt::formatter<Vec3>
{
  constexpr static auto parse(format_parse_context& ctx) { return ctx.begin(); }

  template<typename FormatContext>
  auto format(const Vec3& v, FormatContext& ctx) const
  {
    return format_to(
      ctx.out(), "({:.2f}, {:.2f}, {:.2f})", v.a[0], v.a[1], v.a[2]);
  }
};

template<>
struct fmt::formatter<Mat3x3>
{
  constexpr static auto parse(format_parse_context& ctx) { return ctx.begin(); }

  template<typename FormatContext>
  auto format(const Mat3x3& m, FormatContext& ctx) const
  {
    return format_to(
      ctx.out(),
      "([{:.2f}, {:.2f}, {:.2f}], [{:.2f}, {:.2f}, {:.2f}], [{:.2f}, {:.2f}, "
      "{:.2f}])",
      m.a[0][0],
      m.a[0][1],
      m.a[0][2],
      m.a[1][0],
      m.a[1][1],
      m.a[1][2],
      m.a[2][0],
      m.a[2][1],
      m.a[2][2]);
  }
};
