INCLUDE_DIRS = -I/home/ubuntu/local/include/tesseract -L/home/ubuntu/local/lib
LIB_DIRS = 
CC=g++

CDEFS=
CFLAGS= -O0 -g -w $(INCLUDE_DIRS) $(CDEFS)
LIBS=
CPPLIBS= -L/usr/lib -lopencv_core -lopencv_flann -lopencv_video -lopencv_legacy -lopencv_highgui -lopencv_imgproc -lopencv_calib3d -lopencv_contrib -lopencv_features2d -lopencv_gpu -lopencv_objdetect -llept -lcudart -ldl -lopencv_tegra -lopencv_superres -lespeak -ltesseract -lrt -lpthread -lm

HFILES= 
CFILES= 
CPPFILES= main.cpp

SRCS= ${HFILES} ${CFILES}
CPPOBJS= ${CPPFILES:.cpp=.o}

all:	 main

clean:
	-rm -f *.o *.d output.txt
	-rm -rf pics
	-rm -f main
	mkdir pics

distclean:
	-rm -f *.o *.d

main: main.o
	$(CC) $(LDFLAGS) $(CFLAGS) -o $@ $@.o `pkg-config --libs opencv` $(CPPLIBS)

depend:

.c.o:
	$(CC) $(CFLAGS) -c $<

.cpp.o:
	$(CC) $(CFLAGS) -c $<
