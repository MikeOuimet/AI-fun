#include <stdio.h>
#include <stdlib.h>
#include <string.h>




int main(int argc, char **argv)
{
	char *key = argv[2];
	char *s = argv[1];
	printf("Plaintext is %s\n", s);
	printf("Vigenere key is %s\n", key);
	printf("Equivalent k is : ");
	int l = strlen(key);
	for (int c =0; c<l; c++){
		if (key[c] >=97){
		printf("%i ", key[c]-'a');
		}
		else{
			printf("%i ", key[c]-'A');
		}

	}
	printf("\n");
	int s_l = strlen(s);
	int k;
	for (int c =0; c<s_l; c++)
	{	
		
		if (key[c%l] >=97){
			k = key[c%l]-'a';
			//printf("%c\n",key[c%l]);
			//printf("Numerical key is %i\n", k);
		}
		else{
			k = key[c%l]-'A';
			//printf("Numerical key is %i\n", k);
		}
		//printf("%i\n", s[c]);
		if (s[c] >= 97){
			s[c] = (s[c] + k- 'a')%26 +'a';
		}

		else {
			s[c] = (s[c] + k- 'A')%26 +'A';
		}
		//printf("%c\n", s[c]+'a');


	}
	printf("Encyphered text is %s\n", s);

	return 0;


}
