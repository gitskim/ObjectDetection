c++ follows procedural programming.
---
* clang++ is better than gcc when it comes to cpp file compilation.
* cpp is a superset of c.
* a valid c program is usually a valid cpp program.

---
Pass-by-value v. Pass-by-ref
1) f(struct Pt p)
2) f(struct pt *p)
3) f(struct Pt &p)


### Namespace
```c
#include <iostream>
int main() {
	std::cout << "Hello World";
}
```

but if name space is declared in the beginning

```c
#include <iostream>

using namespace std;

int main() {
	cout << "Hello World";
}
```
### C++ constructors
Stack-allocated objects
* String s1;
* String s2 {};
* String s3("hello");

* int i1 = 7.1; // i1 becomes 7
* int i2 {7.2}; // error because floaing point to integer conversion
* int i3 = {7.2}; // error because floating point to integer conversion. Also '=' is redundant.
* Narrowing conversions allowed such as double to int and int to char for C compatibility, but not recommended.

Heap-allocated objects
* String *p1 = new String();
* String *a1 = new String [10];

### Pointer and references
* String *p3 = &s3;

* String& r3 = s3; // this is a reference to string s3. So if I change the value of r3, s3's valuel will change too. It's like calling s3 with a different name.

```c
void f(Vector v, Vector &v, Vector *pv) {
	v.sz = 10; // doesn't change the actual value
	rv.sz = 10; // changes the actual value
	pv->sz = 10; // changes the actual value.

}
```
```c
//const MyString& rhs means this rhs is the same as MyString that is sent as a parameter. If & comes next to the name MyString, it means it's the same thing and just called with a different name.

MyString& MyString::operator=(const MyString& rhs) {
	// rhs here address to rhs. normal c thing.
	if (this == &rhs) {
		return *this;
	}

	delete[] data;
	len = rhs.len;
	data = new char[len+1];
	strcpy(data, rhs.data);
}
```
* If any custom constructor is created, you cannot use default construct.
* Default constructor allocates member variables on the stack and move the stack bar up when default destructor is called.
### Variable Scope
Namespace scope
* its scope extends from the point of declaration to the end of its namespace.
* an object created by "new" lives until destroyed by delete.

### Class
```c
class Vector {
	public:
		Vector(int s) :elem{new double[s]}, sz{s} {} //construct a vector
		double& operator[](int i) {
			return elem[i];} // element access, operator[] is defined to take in int.
		int size() { return sz;}

	private:
		double *elem;
		int sz;
}
```
* Vector(int) defines how objects of type Vector are constructured.
* The constructor initializes the Vector members using a member initalizer list. It means elem with a pointer to double int is initilaized first and sz is initialized.

### Difference between classes and structs
* In a struct: members defined prior to first access specifier like public, private, etc are public by default.
* In a class: members are private.

### Copy constructor
Shallow copy
```c
MyString s = MyString();
MyString s1 = s;
```
It has the block of memory allocated to MyString s1, but everything inside points to s's, like it was forked, but s1 does not have its own.
* 3 cases copy constructors are called:
* MyString s1(s2);
* foo(MyString s1);
* return s2; //because variable

### Function Declaration
#### Inline function
* Every time a function is called, it loads the function into the memory, copies arguemtns, jumps to the memory location of teh called function, executes the function codes, stores the return values and then jumps back to teh address fo teh instruction that was aved, etc. It is too much. When inline is added to the function a a keyword, the compiler replaces the function call with the function code itself.
* Member functinos defined within a class definition are implicitly inline.

#### Out of scope function e.g. MyString::doWhatever()
* Usually longer function goes here. Cuz Inline function is like a macro's copy and paste, you want to put a shorter function in inline and longer function here.

### Difference between new and malloc()
The following blocks of code do the same thing
* malloc
```c
Pt *p2 = malloc(sizeof(Pt));
p2->x = 4;
p2->y = 4;
```
* new
```c
Pt *p2 = new Pt(4, 4);
```

What about an array?
```c
Pt *myarray = new Pt[10];
```
