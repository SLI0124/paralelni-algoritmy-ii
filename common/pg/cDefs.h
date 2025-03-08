#ifndef __CDEFS_H_
#define __CDEFS_H_

#define __DEBUG_PRINT_

#define FLOAT_EPSILON 0.0001

//#define SWAP(a, b) {a ^= b; b ^= a; a ^= b;}
#define SWAP(a, b) (a^=b^=a^=b)
#define MINIMUM(a, b) ((a) < (b) ? (a) : (b))
#define MAXIMUM(a, b) ((a) > (b) ? (a) : (b))
#define CLIP(a, b, c) MINIMUM(MAXIMUM((a), (b)), (c))					//a = your value, b = left bound, c = rightbound
#define INTERPOLATE(first,last,x) ((x-first)/(last-first))
#define ISININTERVAL(first,last,x) ((first<=x)&&(x<=last))
#define NOTININTERVAL(first,last,x) ((x<first)||(last<x))
#define CHECK_ZERO(x) ((x < FLOAT_EPSILON) || (-x > FLOAT_EPSILON))

#define SAFE_DELETE(p) if(p){delete (p);(p)=nullptr;}
#define SAFE_DELETE_ARRAY(p) if(p){delete[] (p);(p)=nullptr;}

#define GET_K_BIT(n, k) ((n >> k) & 1)


static unsigned int nextPow2(unsigned int x)
{
        --x;
        x |= x >> 1;
        x |= x >> 2;
        x |= x >> 4;
        x |= x >> 8;
        x |= x >> 16;
        return ++x;
}


#endif