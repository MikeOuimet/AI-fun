#include <stdio.h>

int numPlaces(unsigned long long int n)
{
    int r = 1;
    while (n > 9) {
        n = n/10;
        r++;
    }
    return r;
}

int sumevens(unsigned long long int number)
{
    int sum = 0;
    int val;
    while(number >0) {
        val = (number /10) %10;
        if (2*val<9) {
            sum += 2*val;
        }
        else {
            sum = sum+ (2*val)/10 +(2*val)%10;
        }
        number /= 100;
    }
    return sum;
}

int sumodds(unsigned long long int number)
{
    int sum = 0;
    int val;
    while(number >0) {
        val = number%10;
        sum += val;
        number /= 100;
    }
    return sum;
}

int main(void)
{
    unsigned long long int number = 391447635398431;
    printf("The credit card number is %llu\n", number);
    int esum = sumevens(number);
    int osum = sumodds(number);
    int checksum = (esum+osum)%10;
    if (checksum == 0) {
        if ((numPlaces(number)==16) && (number/1000000000000000==4)){
          printf("Valid credit card: Visa\n");  
        }
        else if ((numPlaces(number)==16) && (number/1000000000000000==5) && ((number/100000000000000)%10 < 6) ) {
          printf("Valid credit card: MasterCard\n");  
        }
        else if ((numPlaces(number)==15) && (number/100000000000000==3) && ( ((number/10000000000000)%10 ==4)  || ((number/10000000000000)%10 ==7)  ) ) {
          printf("Valid credit card: American Express\n");  
        }
        else {
            printf("Valid credit card: Unknown bank\n"); 
        }
        
    }
    else {
        printf("Invalid credit card\n");
    }
    return 0;
}