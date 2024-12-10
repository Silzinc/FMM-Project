#include "sample.hpp"
#include "utils.hpp"
#include <optional>

MassSample::MassSample(
  Vec3 pos,
  const std::optional<Vec3> prev_pos = std::nullopt,
  const double mass = 1.0f)
  : mass(mass)
  , position(pos)
  , prev_pos(prev_pos.value_or(pos))
{}

Vec3
MassSample::speed(const double dt) const
{
  return (position - prev_pos) / dt;
}
