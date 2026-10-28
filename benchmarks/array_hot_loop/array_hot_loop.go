package main

import (
	"fmt"
	"os"
	"strconv"
)

func arg(i int) int64 {
	v, err := strconv.ParseInt(os.Args[i], 10, 64)
	if err != nil {
		panic(err)
	}
	return v
}

func main() {
	n := arg(1)
	passes := arg(2)
	arr := make([]uint64, n)
	for i := range arr {
		arr[i] = uint64(i)*7 + 3
	}

	var total uint64
	for pass := int64(0); pass < passes; pass++ {
		scale := uint64(pass) * 3
		for _, x := range arr {
			total += x * scale
		}
	}
	fmt.Println(total)
}
