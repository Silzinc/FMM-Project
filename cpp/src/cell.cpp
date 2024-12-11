#include "cell.hpp"

namespace qvm = boost::qvm;

namespace fmm
{
FieldTensor::FieldTensor(const Vec3& field, const Mat3x3& jacobian)
  : field(field)
  , jacobian(jacobian)
{}

void
FieldTensor::clear()
{
  field *= 0.0;
  jacobian *= 0.0;
}

void
FieldTensor::operator+=(const FieldTensor& rhs)
{
  field += rhs.field;
  jacobian += rhs.jacobian;
}

FieldTensor
FieldTensor::operator+(const FieldTensor& rhs) const
{
  return { field + rhs.field, jacobian + rhs.jacobian };
}

void
FieldTensor::operator-=(const FieldTensor& rhs)
{
  field -= rhs.field;
  jacobian -= rhs.jacobian;
}

FieldTensor
FieldTensor::operator-(const FieldTensor& rhs) const
{
  return { field - rhs.field, jacobian - rhs.jacobian };
}

FMMCell::FMMCell(const Vec3& centroid, const double size)
  : centroid(centroid)
  , size(size)
  // clang-format off
  , field_tensor({
      { 0.0, 0.0, 0.0 },
      { 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0,
        0.0, 0.0, 0.0 }
  })
  // clang-format on
  , mass(0.0)
  , barycenter({ 0.0, 0.0, 0.0 })
  , interaction_list({})
  , direct_neighbors({})
{}

bool
FMMCell::contains_sample(const MassSample& sample) const
{
  const auto diff = sample.position - centroid;
  return std::max(
           { std::abs(diff.a[0]), std::abs(diff.a[1]), std::abs(diff.a[2]) }
         ) < size / 2.0;
}
}
