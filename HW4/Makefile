all: zip

PART1_FILES = part1/hello.cc part1/pi_block_linear.cc part1/pi_block_tree.cc part1/pi_gather.cc part1/pi_nonblock_linear.cc part1/pi_one_side.cc part1/pi_reduce.cc
PART2_FILES = part2/Makefile part2/main.cc part2/matmul.cpp

zip: $(PART1_FILES) $(PART2_FILES) url.txt
	zip HW4_310552020.zip $^

clean:
	rm -f HW4_310552020.zip