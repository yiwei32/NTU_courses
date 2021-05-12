#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

// Create a XOR linked list
/*
    support forward and backward traverse
    A <-> B <-> C-> NULL
    since npx = add(prev) XOR add(next)
    add(B) = A->npx XOR NULL
    add(C) = B->npx XOR add(A)
*/

typedef struct Node{
    int data;
    struct Node* npx; // npx = add(prev) XOR add(next)
}Node;

typedef struct XList{
    Node* head;
    Node* tail;
    unsigned size;

}XList;

Node* XOR(Node* add1, Node* add2);
XList* CreateXList(void);
Node* CreateNode(int data);
int isEmpty(XList *list);
void insertNodeTail(XList *list, int data);
void deleteNodeTail(XList *list);
void migrate(XList *list_from, XList *list_to);
void printXList(XList *list);

int ReadInt(void){
    char digit = getchar();
    int value = 0;
    while(digit >= '0' && digit <= '9'){
        value = 10 * value + (digit - '0');
        digit = getchar();
    }
    return value;
};

int main(){
    int k, n;
    k = ReadInt();
    n = ReadInt();

    XList* rail[k];
    for (int i = 0; i < k; i++){
        rail[i] = CreateXList();
    } 

    // read records and do operation accordingly
    

    for (int i = 0; i < n; i++){

        char command[7];
        int first;
        int second;
        char digit = getchar();
        while (getchar() != ' ');
        
        if (digit == 'e'){
            first = ReadInt();
            second = ReadInt(); 
            insertNodeTail(rail[first], second);
        }
        else if (digit == 'l'){
            first = ReadInt();
            deleteNodeTail(rail[first]);
        }
        else if (digit == 'm'){
            first = ReadInt();
            second = ReadInt(); 
            migrate(rail[first], rail[second]);
        }
    }

    // print results
    for (int i = 0; i < k; i++){
        printXList(rail[i]);
    }

    return 0;
};

Node* XOR(Node* add1, Node* add2){
    return (Node*)((intptr_t)add1 ^ (intptr_t)add2);
};

XList* CreateXList(void){
    XList *newList = malloc(sizeof(XList));
    if (newList !=NULL){
        newList->head = NULL;
        newList->tail = NULL;
        newList->size = 0;
    }
    return newList;
};

Node* CreateNode(int data){
    Node *newNode = malloc(sizeof(Node));
    if (newNode != NULL){
        newNode->data = data;
        newNode->npx = NULL;
    }
    return newNode;
};

int isEmpty(XList *list){
    return list->size == 0;
};

void insertNodeTail(XList *list, int data){
    Node *newNode = CreateNode(data);
    if (newNode != NULL){
        if (!isEmpty(list)){
            list->tail->npx = XOR(list->tail->npx, newNode);
            newNode->npx = list->tail;
            list->tail = newNode;
        }
        else{
            list->head = list->tail = newNode;
        }
        list->size++;
    }
    
};

void deleteNodeTail(XList *list){
    if (!isEmpty(list)){
        if (list->size == 1){
            free(list->head);
            list->head = NULL;
            list->tail = NULL;
        }
        else{
            Node *preTail = list->tail->npx;
            preTail->npx = XOR(preTail->npx, list->tail);
            free(list->tail);
            list->tail = preTail;
            
        }
        list->size--;
    }
    return;
    // if list is empty, do nothing.
};

void migrate(XList *list_from, XList *list_to){
    if (isEmpty(list_from)){
        return;
    }
    else if (isEmpty(list_to)){
        list_to->head = list_from->tail;
        list_to->tail = list_from->head;
        list_from->head = list_from->tail = NULL;
        list_to->size = list_from->size;
        list_from->size = 0;
    }
    else{
        // tail to tail, update two tail's npx
        list_to->tail->npx = XOR(list_to->tail->npx, list_from->tail);
        list_from->tail->npx = XOR(list_from->tail->npx, list_to->tail);
        list_to->tail = list_from->head;
        list_from->head = list_from->tail = NULL;
        list_to->size += list_from->size;
        list_from->size = 0;
    }
};

void printXList(XList *list){
    Node *current = list->head;
    Node *prev = NULL;
    while (current != NULL){
        printf("%d ",current->data);
        Node *next = XOR(current->npx, prev);
        prev = current;
        current = next;
    }
    printf("\n");
};
