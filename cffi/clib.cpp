#include<iostream>


extern "C"
{
void resize()
{
    using namespace std;
    cout << "in the lib" << endl;
}

}
