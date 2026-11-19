// Writes each row's list (uint32, row order) of a saved TieredQIVFIndex: the shared #55 partition for SAQ.
#include <fstream>
#include <iostream>

#include "tieredq/tieredq.h"

int main(int argc, char** argv) {
  if (argc != 3) {
    std::cerr << "usage: export_assignment INDEX.tq OUT.u32\n";
    return 2;
  }
  const auto lists = tq::TieredQIVFIndex::load(argv[1]).list_assignment();
  std::ofstream(argv[2], std::ios::binary)
      .write(reinterpret_cast<const char*>(lists.data()), std::streamsize(lists.size() * sizeof(lists[0])));
  return 0;
}
