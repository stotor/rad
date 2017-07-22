CC = gcc

CCFLAGS = -shared -fPIC -O3 -Wall

SOURCES = create_radiograph.c

create_radiograph.so : $(SOURCES)
	$(CC) $(CCFLAGS) $(SOURCES) -o $@

clean :
	rm create_radiograph.so
