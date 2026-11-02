#pragma once
// Emscripten has no sendfile, which vkdt's fs_copy_file uses.
#include <sys/types.h>
#include <unistd.h>

static inline ssize_t sendfile(int out_fd, int in_fd, off_t *offset, size_t count) {
  char buf[65536];
  if(offset) lseek(in_fd, *offset, SEEK_SET);
  ssize_t total = 0;
  while(count > 0) {
    ssize_t n = read(in_fd, buf, count < sizeof(buf) ? count : sizeof(buf));
    if(n <= 0) break;
    if(write(out_fd, buf, n) != n) return -1;
    total += n;
    count -= n;
  }
  if(offset) *offset += total;
  return total;
}
