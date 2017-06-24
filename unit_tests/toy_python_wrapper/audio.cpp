#include <iostream>

#include "audio.h"

class A
{
public:
	A(void){i = 1;};
	void fn(void){std::cout << "A.i = " << i++ << std::endl;};
private:
	int i;
};

static A a=A();

void playNext(void)
{	
	std::cout << "C++ playNext() was called" << std::endl;
	a.fn();
}
