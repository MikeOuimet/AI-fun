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
		printf("%i ", key[c]-'a');

	}
	printf("\n");
	int s_l = strlen(s);
	int k;
	for (int c =0; c<s_l; c++)
	{	
		

		k = key[c%l]-'a';
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
