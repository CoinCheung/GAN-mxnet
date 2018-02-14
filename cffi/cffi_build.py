#!/usr/bin/python


from cffi import FFI
import os


ffibuild = FFI()


ffibuild.set_source("cwrapper", r'''
                    #include "clib.h"

                    void resize_wrap()
                    {
                        resize();
                    }

                    ''',
                    libraries=['clib'],
                    library_dirs=['./'],
                    include_dirs=['./'],
                    extra_link_args=['-Wl,-rpath=./cffi/']
)


ffibuild.cdef(r'''
              void resize_wrap();
              '''

)


if __name__ == "__main__":
    ffibuild.compile(verbose=True, debug=False)

    filename = ["./cwrapper.o","./cwrapper.c"]

    for f in filename:
        os.shutil.move(f, )



