//
// Created by napbad on 12/12/25.
//

#include <gtest/gtest.h>

int main()
{
    ::testing::InitGoogleTest();
    return RUN_ALL_TESTS();
}


// #include <initializer_list>
// #include <iostream>
// #include <ostream>

// void printRecursive(int a) {
//     std::cout << a << ", ";
// }

// template<typename T>
// void printRecursive(std::initializer_list<T> list);

// template<typename T>
// void printRecursive(std::initializer_list<T> list) {
//     std::cout << "{ ";
//     for (auto &t : list) {
//         printRecursive(t);
//     }
//     std::cout << "} \n";
// }

// template<typename T>
// void printRecursive(std::initializer_list<T> list) {
//     std::cout << "{ ";
//     for (const auto &t : list) {
//         printRecursive(t);
//     }
//     std::cout << "} ";
// }


// // This version handles ANY nesting depth
// template<typename T>
// void printRecursive(const T& container) {  // Note: const T& instead of initializer_list directly
//     if constexpr (requires { container.begin(); container.end(); }) {  // C++20: check if iterable
//         std::cout << "{ ";
//         for (const auto& elem : container) {
//             printRecursive(elem);
//         }
//         std::cout << "} ";
//     } else {
//         std::cout << container << ", ";  // fallback, but we rely on int overload
//     }
// }

// int main() {

//     printRecursive({1, 2, 3, 4});

//     printRecursive({{1, 2, 3, 4}, {1, 2, 3, 4}});

//     return 0;
// }