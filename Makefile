CPP = g++
CFLAGS = $(shell pkg-config --cflags opencv)  -c -fPIC
LFLAGS = $(shell pkg-config --libs opencv) -shared
SRCDIR = ./cffi
ODIR = ./cffi/build
LIBDIR = ./cffi
_OBJS = clib.o 
_SO = clib

OBJS = $(patsubst %.o, $(ODIR)/%.o, $(_OBJS))
SO = $(patsubst %, $(LIBDIR)/lib%.so, $(_SO))

	

$(SO): $(OBJS)
	$(CPP) $(LFLAGS) $^ -o $@
	cd cffi && python cffi_build.py 

$(ODIR)/%.o: $(SRCDIR)/%.cpp
	@mkdir -p $(ODIR)
	$(CPP) $(CFLAGS) $^ -o $@



clean:
	rm -rf $(ODIR) $(LIBDIR)/*so




