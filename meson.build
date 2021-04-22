project('disc', 'cpp',  default_options : ['cpp_std=c++17', 'buildtype=release'])

boost = dependency('boost', version : '>=1.5.0')
openmp = dependency('openmp')
quadmath = dependency('quadmath', required: false)
pybind11 = dependency('pybind11', required: false)

cc = meson.get_compiler('cpp')

args = []
deps = [openmp, boost]

if cc.has_header('execution')
    args += ['-DHAS_EXECUTION_POLICIES']
    tbb = dependency('tbb')
    deps += [tbb]
endif

if quadmath.found() or 'gcc' == cc.get_argument_syntax()
    args += ['-DWITH_QUADMATH', '-std=gnu++17', '-fext-numeric-literals']
elif mpfr.found()
    args += ['-DWITH_MPFR']
endif


disc = declare_dependency(
    include_directories : ['./include'], 
    dependencies : deps, 
    compile_args: args)

if pybind11.found()
    pymod = import('python')
    py = pymod.find_installation('python3')

    py.extension_module('disc', 
        sources : ['./src/bindings/python/PyDisc.cpp'], 
        include_directories : ['./src'], 
        dependencies : [disc, py.dependency(), pybind11],
        install : true
    )
endif
