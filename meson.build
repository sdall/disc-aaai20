project('disc', 'cpp',  default_options : ['cpp_std=c++17', 'buildtype=release'])

boost = dependency('boost', version : '>=1.5.0')
tbb = dependency('tbb')
openmp = dependency('openmp')
quadmath = dependency('quadmath', required: false)

cc = meson.get_compiler('cpp')
args = ['-DWITH_EXECUTION_POLICIES']
if quadmath.found() or 'gcc' == cc.get_argument_syntax()
    args += ['-DWITH_QUADMATH', '-std=gnu++17']
endif

disc = declare_dependency(
    include_directories : ['./include'], 
    dependencies : [boost, tbb, openmp, quadmath], 
    compile_args: args)

pybind11 = dependency('pybind11', required: false)

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
