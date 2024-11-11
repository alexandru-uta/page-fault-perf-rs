#include <stddef.h>
#include <stdio.h>
#include <sys/mman.h>
#include <string.h>

#define SMALL_PAGE_SIZE 4096
#define LARGE_PAGE_SIZE (2 * 1024 * 1024)
#define WASM_PAGE_SIZE 65536

int main(int argc, char *argv[]) {

  // Create an address space of 2MB.
  char *addr = (char *)mmap(0, LARGE_PAGE_SIZE, PROT_WRITE | PROT_READ,
                            MAP_SHARED | MAP_ANONYMOUS, -1, 0);

  // Set the region to use transparent huge pages.
  //madvise(addr, LARGE_PAGE_SIZE, MADV_HUGEPAGE);
  // Fill the region with some value.
  memset(addr, 13, LARGE_PAGE_SIZE);

  // Protect the region beyond WASM_PAGE_SIZE.
  //int res = mprotect(addr + WASM_PAGE_SIZE, LARGE_PAGE_SIZE - WASM_PAGE_SIZE, PROT_NONE);
  //printf("mprotect result: %d\n", res);

  // Touch a page within the protected region.
  // This should cause a segfault.
  addr[WASM_PAGE_SIZE + 100] = 42;

  munmap((void *)addr, LARGE_PAGE_SIZE);

  return 0;
}
