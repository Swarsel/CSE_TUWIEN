/// \file
/// \brief List class declaration

#pragma once

#include <cstddef>

namespace ex13 {

class List {
public:
  using value_type = double;
  using size_type = std::size_t;

  /// \Brief nested class representing a node in the list
  struct Node {
    value_type value;        ///< stored value
    Node* next = nullptr;    ///< pointer to next node in list

    /// \brief Custom constructor
    Node(const value_type&);
  };

private:
  Node* _head = nullptr; ///< root element
  size_type _size = 0;   ///< current number of nodes in list

public:
  /// \brief Default constructor
  List();

  /// \brief Custom constructor
  /// \param n number of elements to create
  /// \param init initial value for all elements
  List(size_type n, const value_type& init);

  /// \brief Destructor
  ~List();

  /// \brief Copy constructor
  /// \param other object to be copied
  List(const List& other);

  /// \brief Move constructor
  /// \param other object to be moved from
  List(List&& other); // move ctor

  /// \brief Copy assignment operator
  /// \param other object to be copied from
  List& operator=(const List& other);

  /// \brief Move assignment operator
  /// \param other object to be moved from
  List& operator=(List&& other);

  /// \brief Getter for _size
  size_type size() const;

  /// \brief Getter for _head
  const Node* data() const;

  /// \brief Access front element
  value_type& front();

  /// \brief Access front element (read-only)
  const value_type& front() const;

  /// \brief Push new list element at the front
  void push_front(const value_type&);

  /// \brief Pop element at the front
  void pop_front();
};

} // namespace ex13
