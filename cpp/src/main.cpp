#include "boost/multi_array.hpp"
#include <cassert>
#include <fmt/core.h>

struct Foo
{
  int x;
  int y;
  Foo(int x = 1, int y = 0) : x(x), y(y)
  {
  }

  bool operator==(const Foo& rhs) const
  {
    return x == rhs.x && y == rhs.y;
  }
};

int main()
{
  // Create a 3D array that is 3 x 4 x 2
  typedef boost::multi_array<Foo, 3> array_type;
  typedef array_type::index index;
  array_type A(boost::extents[3][4][2]);

  // Assign values to the elements
  int values = 0;
  for (index i = 0; i != 3; ++i)
    for (index j = 0; j != 4; ++j)
      for (index k = 0; k != 2; ++k)
      {
        A[i][j][k] = Foo(values, values + 1);
        values++;
      }

  // Verify values
  int verify = 0;
  for (index i = 0; i != 3; ++i)
    for (index j = 0; j != 4; ++j)
      for (index k = 0; k != 2; ++k)
      {
        assert(A[i][j][k] == Foo(verify, verify + 1));
        verify++;
      }

  fmt::print("Hello World!\n");
  return 0;
}
