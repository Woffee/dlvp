#include <iostream>

using namespace std;

int funcD(){
	return 0;
}


int funcC(){
	funcD();
	return 0;
}

int funcB(){
	funcC();
	return 0;
}


int funcA(){
	funcB();
	return 0;
}


int main(){
	funcA();
}