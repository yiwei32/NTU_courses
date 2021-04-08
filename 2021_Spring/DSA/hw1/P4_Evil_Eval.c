#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define MAX 1000000


// Struct char stack
struct Stack_char{
    int top;
    int capacity;
    char* arr;
};

struct Stack_char* CreateCharStack(int capacity){
    struct Stack_char *stack = (struct Stack_char*)malloc(sizeof(struct Stack_char));
    stack->capacity = capacity;
    stack->top = -1;
    stack->arr = (char*)malloc(sizeof(char) * capacity);
    return stack;
};

int stack_char_isFull(struct Stack_char* stack){
    return stack->top == stack->capacity - 1;
};

int stack_char_isEmpty(struct Stack_char* stack){
    return stack->top == -1;
};

void stack_char_push(struct Stack_char* stack, char token){
    if (!stack_char_isFull(stack)){
        stack->arr[++stack->top] = token;
        // printf("Push %c to stack\n", token);
    }
    else{
        printf("Overflow\nProgram Terminated\n");
        exit(EXIT_FAILURE);
    }
};

char stack_char_pop(struct Stack_char* stack){
    if (!stack_char_isEmpty(stack)){
        return stack->arr[stack->top--];
        
    }
    else{
        printf("Stack_char is empty when pop\n");
        exit(EXIT_FAILURE);
    }
};

char stack_char_peek(struct Stack_char* stack){
    if(!stack_char_isEmpty(stack)){
        return stack->arr[stack->top];
        
    }
    else{
        printf("Stack_char is empty when peek\n");
        exit(EXIT_FAILURE);
    }
}

void stack_char_free(struct Stack_char* stack){
    free(stack->arr);
    free(stack);
};


// Struct double stack

struct Stack_double{
    int top;
    int capacity;
    double* arr;
};

struct Stack_double* CreateDoubleStack(int capacity){
    struct Stack_double *stack = (struct Stack_double*)malloc(sizeof(struct Stack_double));
    stack->capacity = capacity;
    stack->top = -1;
    stack->arr = (double*)malloc(sizeof(double) * capacity);
    return stack;
};

int stack_double_isFull(struct Stack_double* stack){
    return stack->top == stack->capacity - 1;
};

int stack_double_isEmpty(struct Stack_double* stack){
    return stack->top == -1;
};

void stack_double_push(struct Stack_double* stack, double token){
    if (!stack_double_isFull(stack)){
        stack->arr[++stack->top] = token;
        // printf("Push %.2lf to stack\n", token);
    }
    else{
        printf("Overflow\nProgram Terminated\n");
        exit(EXIT_FAILURE);
    }
};

double stack_double_pop(struct Stack_double* stack){
    if (!stack_double_isEmpty(stack)){
        return stack->arr[stack->top--];
        
    }
    else{
        printf("Stack_double is empty when pop\n");
        exit(EXIT_FAILURE);
    }
};

double stack_double_peek(struct Stack_double* stack){
    if(!stack_double_isEmpty(stack)){
        return stack->arr[stack->top];
        
    }
    else{
        printf("Stack_double is empty when peek\n");
        exit(EXIT_FAILURE);
    }
}

void stack_double_free(struct Stack_double* stack){
    free(stack->arr);
    free(stack);
};



// some Utility functions

int isDigit(char token){
    if (token >= '0' && token <= '9'){
        return 1;
    }
    else{
        return 0;
    }
};

int precedence(char token){
    if (token == '/' || token == '*'){
        return 2;
    }
    else if (token == '+' || token == '-'){
        return 1;
    }
    return 0;
}

double applyOp(double val_1, double val_2, char op){
    switch (op){
        case '+':
            return val_1 + val_2;
        case '-':
            return val_1 - val_2;
        case '*':
            return val_1 * val_2;
        case '/':
            return val_1 / val_2;
    }
    printf("Wrong operator!\n");
    exit(EXIT_FAILURE);
}


double Eval(char* tokens){
    
    int length = strlen(tokens);
    struct Stack_double* ValueStack = CreateDoubleStack(length);
    struct Stack_char* OpStack = CreateCharStack(length);

    for (int i = 0; i < length; i++){
        /*  
        for digits, check negative numbers
        i.e. check if there is a number before '-'
        */
        if (tokens[i] == '-' && (tokens[i-1] == '(' || i == 0)){
            double value = 0;
            i++;
            while(i < length && isDigit(tokens[i])){
                value = value*10 + (double)(tokens[i] - '0');
                i++;
            }
            i--; //correct the index back to previous one
            stack_double_push(ValueStack, -value);
        }
        else if (isDigit(tokens[i])){
            double value = 0;
            // check continuous digits 
            while(i < length && isDigit(tokens[i])){
                value = value*10 + (double)(tokens[i] - '0');
                i++;
            }
            i--; // coreect the index back to previous one
            stack_double_push(ValueStack, value);
        }
        else if (tokens[i] == '('){
            stack_char_push(OpStack, tokens[i]);
        }
        // solve entire brace when encountering a closing brace
        else if (tokens[i] == ')'){
            while(!stack_char_isEmpty(OpStack) && stack_char_peek(OpStack) != '('){
                char op = stack_char_pop(OpStack);
                double val_2 = stack_double_pop(ValueStack);
                double val_1 = stack_double_pop(ValueStack);
                stack_double_push(ValueStack, applyOp(val_1, val_2, op));
            }
            if (stack_char_peek(OpStack) == '('){
                char temp = stack_char_pop(OpStack); // pop a garbage
            }
        }
        // Current token is an operator
        else{
            /* 
            while the top of Opstack has the same or greater precedence than the current token,
            pop it and two values, calculate the result
            */
            int Prec_token = precedence(tokens[i]);
            while(!stack_char_isEmpty(OpStack) && precedence(stack_char_peek(OpStack)) >= Prec_token){
                char op = stack_char_pop(OpStack);
                double val_2 = stack_double_pop(ValueStack);
                double val_1 = stack_double_pop(ValueStack);
                stack_double_push(ValueStack, applyOp(val_1, val_2, op));
            }
            // push current token to stack
            stack_char_push(OpStack, tokens[i]);
        }
    }
    // Now all tokens are scanned, calculate the remaining things in stack
    while(!stack_char_isEmpty(OpStack)){
        char op = stack_char_pop(OpStack);
        double val_2 = stack_double_pop(ValueStack);
        double val_1 = stack_double_pop(ValueStack);
        stack_double_push(ValueStack, applyOp(val_1, val_2, op));
    }

    double ans = stack_double_pop(ValueStack);
    stack_double_free(ValueStack);
    stack_char_free(OpStack);
    return ans;
};

int main(){
    
    char c ;
    char tokens[MAX];
    int index = 0;
    while(scanf("%s", tokens) != EOF){
        double ans = Eval(tokens);
        printf("%.14lf\n", ans);
        index = 0;
    }
    
    return 0;
}
