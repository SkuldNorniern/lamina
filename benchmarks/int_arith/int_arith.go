package main

import (
	"fmt"
	"os"
	"strconv"
)

//go:noinline
func arg(i int, fallback int64) int64 {
	if i >= len(os.Args) {
		return fallback
	}
	v, err := strconv.ParseInt(os.Args[i], 10, 64)
	if err != nil {
		panic(err)
	}
	return v
}

func main() {
	iters := arg(1, 100000000)
	x := uint64(arg(2, 12345))
	var y, z uint64
	for i := int64(0); i < iters; i++ {
		m := x*1103515245 + 12345
		x = m & 2147483647
		s := x >> 7
		s2 := y << 3
		y = (s ^ s2) & 2147483647
		w := y ^ x
		z += w
	}
	fmt.Println(x)
	fmt.Println(y)
	fmt.Println(z)
}
