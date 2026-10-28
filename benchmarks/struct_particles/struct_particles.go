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

type particle struct {
	x, y, vx, vy int64
}

func main() {
	steps := arg(1)
	p := particle{0, 0, arg(2), arg(3)}
	var flips int64
	for i := int64(0); i < steps; i++ {
		p.x += p.vx
		p.y += p.vy
		if p.x*p.x > 1000000 {
			p.vx = -p.vx
			flips++
		}
		if p.y*p.y > 1000000 {
			p.vy = -p.vy
			flips++
		}
	}
	fmt.Println(p.x)
	fmt.Println(p.y)
	fmt.Println(flips)
}
