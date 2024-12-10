#include "cell.hpp"

namespace qvm = boost::qvm;

FieldTensor::FieldTensor(const Vec3& field, const Mat3x3& jacobian)
  : field(field)
  , jacobian(jacobian)
{}

void
FieldTensor::clear()
{
  field *= 0.0f;
  jacobian *= 0.0f;
}

FieldTensor&
FieldTensor::operator+=(const FieldTensor& rhs)
{
  field += rhs.field;
  jacobian += rhs.jacobian;
  return *this;
}

FieldTensor
FieldTensor::operator+(const FieldTensor& rhs) const
{
  FieldTensor result = *this;
  result += rhs;
  return result;
}

FieldTensor&
FieldTensor::operator-=(const FieldTensor& rhs)
{
  field -= rhs.field;
  jacobian -= rhs.jacobian;
  return *this;
}

FieldTensor
FieldTensor::operator-(const FieldTensor& rhs) const
{
  FieldTensor result = *this;
  result -= rhs;
  return result;
}

FMMCell::FMMCell(const Vec3& centroid, const double size)
  : centroid(centroid)
  , size(size)
  , field_tensor(FieldTensor(
      Vec3{ 0.0f, 0.0f, 0.0f },
      Mat3x3{ 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f }))
  , mass(0.0f)
  , barycenter(Vec3{ 0.0f, 0.0f, 0.0f })
{}

bool
FMMCell::contains_sample(const MassSample& sample) const
{
  return qvm::mag(centroid - sample.position) < size / 2.0f;
}
