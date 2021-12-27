all: copyfiles zip

copyfiles: OpenCL/kernel.cl OpenCL/hostFE.c
	cp $^ ./

COPY_FILES = kernel.cl hostFE.c
zip: $(COPY_FILES) url.txt
	zip HW6_310552020.zip $^

clean:
	rm -f $(COPY_FILES)