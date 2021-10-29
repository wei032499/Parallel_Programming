TARGET := pi.out
CXX = g++ -m64
CXXFLAGS = -O3 -mavx -std=c++17 -Wall -pthread

all: pi.cpp
	$(CXX) $(CXXFLAGS) pi.cpp -o $(TARGET)

clean:
	rm -f *.o *.s $(TARGET) *~