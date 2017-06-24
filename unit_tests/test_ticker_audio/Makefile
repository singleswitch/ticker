all: fmod_alphabet_player.o test_alphabet_player.o test_alphabet_player

CXXFLAGS=-g -O3

VPATH = ./:../../

ifndef TICKER_FMOD_ROOT_DIR
  $(info Environment variable TICKER_FMOD_ROOT_DIR not set. Standard include and library paths will be searched for FMOD instead)
  INCLUDES=-I ../../
  LIBFMOD="-lfmodex.so"
else
  $(info Using environment TICKER_FMOD_ROOT_DIR with value ${TICKER_FMOD_ROOT_DIR})
  INCLUDES=-I ../../ -I ${TICKER_FMOD_ROOT_DIR}/api/inc/
  LIBFMOD=${TICKER_FMOD_ROOT_DIR}/api/lib/libfmodex64.so
endif

fmod_alphabet_player.o: fmod_alphabet_player.cpp fmod_alphabet_player.h 
	g++ ${CCFLAGS} ${INCLUDES} -c $<

test_alphabet_player.o: test_alphabet_player.cpp  
	g++ ${CCFLAGS} ${INCLUDES} -c test_alphabet_player.cpp
 

test_alphabet_player: test_alphabet_player.o fmod_alphabet_player.o
	g++ ${CCFLAGS} -o test_alphabet_player ${INCLUDES} test_alphabet_player.o fmod_alphabet_player.o ${LIBFMOD} -pthread

run:
	./test_alphabet_player

clean:
	rm -f fmod_alphabet_player.o
	rm -f test_alphabet_player.o
	rm test_alphabet_player
