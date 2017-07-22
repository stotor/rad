CXX = g++

CCFLAGS = -shared -fPIC -O3 -Wall

SOURCES = create_radiograph.cpp

create_radiograph.so : $(SOURCES)
	$(CC) $(CCFLAGS) $(SOURCES) -o $@

clean :
	rm create_radiograph.so
