#include <stdio.h>
int main() {
    long long header;
    asm("movz %0, #52501, lsl #0" : "=r"(header));
    printf("After movz: %lld\n", header);
    asm("movk %0, #1883, lsl #16" : "+r"(header));
    printf("After movk: %lld\n", header);
    return 0;
}
