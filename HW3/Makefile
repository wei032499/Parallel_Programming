all: copyfiles zip

copyfiles: part1/cg_impl.c part2/breadth_first_search/bfs.cpp part2/page_rank/page_rank.cpp
	cp $^ ./

COPY_FILES = cg_impl.c bfs.cpp page_rank.cpp
zip: $(COPY_FILES)
	zip HW3_310552020.zip $^

clean:
	rm $(COPY_FILES)