/// \file
/// \brief doubly-linked list with iterator interface

#pragma once

#include <cstddef> // std::size_t
#include <utility> // std::swap
#include <iterator>
#include <type_traits>



/// \todo add standard library headers as needed

namespace ex31 {

/// \brief Fully functional doubly-linked list
/// \todo extend this class by adding a stdlib compatible iterator interface
template <class T> class List {

public:
  using size_type = std::size_t;
  using value_type = T;

private:
  /// \brief Represents the connectivity of a node
  struct Node {
    Node* prev; ///< pointer to previous node in list
    Node* next; ///< pointer to next node in list
  };

  /// \brief Represents the connectivity (base class) and the value of a node
  struct ValueNode : Node {
    T value;
  };

  Node* entry;    ///< entry node (always present)
  size_type size; ///< current number of nodes (excluding the entry node)

  /// \brief swap two lists by swapping both members
  void swap(List& other) {
    std::swap(entry, other.entry);
    std::swap(size, other.size);
  }

  /// \brief Erase a node in the list and rebuild connectivity
  void erase(Node* node) {
    node->prev->next = node->next;
    node->next->prev = node->prev;
    delete static_cast<ValueNode*>(node);
  }

  /// \brief Append the values of another list a the end
  void append(const List& other) {
    for (const auto& item : other) {
      push_back(item);
    }
  }

public:
  /// \brief Remove all value holding nodes in the list
  void clear() {
    auto iter = entry->next;
    while (iter != entry) {
      const auto next = iter->next;
      erase(iter);
      iter = next;
    }
  }

  /// \brief Insert a value at the end of the list
  void push_back(const value_type& value) {
    const auto new_node = new ValueNode{{entry->prev, entry}, value};
    entry->prev->next = new_node;
    entry->prev = new_node;
    ++size;
  }

  /// \brief Insert a value at the the front of th elist
  void push_front(const value_type& value) {
    const auto new_node = new ValueNode{{entry, entry->next}, value};
    entry->next->prev = new_node;
    entry->next = new_node;
    ++size;
  }

  /// \brief Construct an empty list
  List() : size(0) {
    entry = new Node{NULL, NULL};
    entry->prev = entry;
    entry->next = entry;
  };

  /// \brief Construct a list as copy from another list
  List(const List& other) : size(0) {
    entry = new Node{NULL, NULL};
    entry->prev = entry;
    entry->next = entry;
    append(other);
  }

  /// \brief Construct a list by moving-from another list
  List(List&& other) : List() { this->swap(other); }

  /// \brief Assign values from another list
  List& operator=(const List& other) {
    clear();
    append();
    return *this;
  }

  /// \brief Move-Assign values from another list
  List& operator=(List&& other) {
    this->swap(other);
    return *this;
  }

  /// \brief Destruct all nodes in the list
  ~List() {
    clear();
    delete entry;
  }

  template <typename IteratorType, bool IsConst> class baseIterator {
  public:
    using iterator_category = std::bidirectional_iterator_tag;
    using difference_type = std::ptrdiff_t;
    using value_type = std::conditional_t<IsConst, IteratorType, std::remove_const_t<IteratorType>>;
    using pointer = IteratorType*;
    using reference = IteratorType&;

    ValueNode* nodePtr;

    baseIterator(Node *ptr) : nodePtr(static_cast<ValueNode*>(ptr)) {}

    value_type &operator*() { return nodePtr->value; }

    baseIterator operator++() {
      nodePtr = static_cast<ValueNode*>(nodePtr->next);
      return *this;
    }
    baseIterator operator--() {
      nodePtr = static_cast<ValueNode*>(nodePtr->prev);
      return *this;
    }

    pointer operator->() { return &nodePtr->value; }

    bool operator==(const baseIterator& other) { return nodePtr == other.nodePtr; }
    bool operator!=(const baseIterator& other) { return !(*this == other); }
};

  typedef baseIterator<T, false> iterator;
  typedef baseIterator<const T, true> const_iterator;
  iterator begin() { return iterator(entry->next); }
  iterator end() { return iterator(entry); }
  const_iterator begin() const { return const_iterator(entry->next); }
  const_iterator end() const { return const_iterator(entry); }
  const_iterator cbegin() const { return const_iterator(entry->next); }
  const_iterator cend() const { return const_iterator(entry); }

};

} // namespace ex31
