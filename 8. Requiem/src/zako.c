// Zako zako. Russian peasant algorithm.
// For 2^32 unsigned integers.

#include <stdio.h>

unsigned int
add(unsigned int a, unsigned int b)
{
    if (!b) return a;
    
    return add(a ^ b, (a & b) << 1);
}

int
main()
{
    unsigned int a, b, result = 0;
    
    printf("Enter an expression (has to be X * Y):\n");
    scanf("%u * %u", &a, &b);

    loop:
    if (b & 1) result = add(result, a);
    a <<= 1;
    b >>= 1;
    if (b) goto loop;

    printf("= %u\n", result);
    return 0;
}
