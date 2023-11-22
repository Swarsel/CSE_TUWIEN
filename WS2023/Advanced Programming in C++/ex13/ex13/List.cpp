/// \file
/// \brief List class (definitions)

#include "List.hpp"
#include <algorithm>
/// \todo Add standard library headers if needed

namespace ex13 {

List::Node::Node(const value_type& val) : value{val} {}

List::List() {}

List::List(size_type count, const value_type& value) {
  for (std::size_t i = 0; i < count; ++i) {
    push_front(value);
  }
}

List::~List() {
  while (_size > 0)
    pop_front();
}

void List::push_front(const value_type& value) {
  Node* next_head = new Node(value);
  next_head->next = _head;
  _head = next_head;
  ++_size;
}

void List::pop_front() {
  if (_head != nullptr) {
    Node* old_head = _head;
    _head = old_head->next;
    delete old_head;
    --_size;
  }
}

List::size_type List::size() const { return _size; }
const List::Node* List::data() const { return _head; }
const List::value_type& List::front() const { return _head->value; }
List::value_type& List::front() { return _head->value; }

/// \todo implement copy constructor
    List::List(const List& other) : _head(new Node(other._head->value)), _size(1) {
        Node* _node = this->_head;
        Node* _goto = other._head->next;
        while (_goto != nullptr) {
            this->_size++;
            _node->next = new Node(_goto->value);
            _node = _node->next;
            _goto = _goto->next;
        }
    }

/// \todo implement move constructor
    List::List(List&& other) : _head(other._head), _size(other._size) {
        other._head = nullptr;
        other._size = 0;
    }
/// \todo implement copy assignment operator
    List &List::operator=(const List &other) {
        this->~List();
        this->_size = other._size;
        this->_head = new Node (other._head->value);

        Node* _node = this->_head;
        Node* _goto = other._head->next;
        while (_goto != nullptr) {
            _node->next = new Node(_goto->value);
            _node = _node->next;
            _goto = _goto->next;
        }
        return *this;
    }

/// \todo implement move assignment operator
List& List::operator=(List&& other){
    this->~List();
    this->_head = nullptr;
    this->_size = 0;
    std::swap(other._size, _size);
    std::swap(other._head, _head);
    return *this;
    }


} // namespace ex13
