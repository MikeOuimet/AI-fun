#include <stdio.h>
#include <stdlib.h>
#include <string.h>




int main(int argc, char **argv)
{
	long k = strtol(argv[2], NULL, 10);
	char *s = argv[1];
	printf("Plaintext is %s\n", s);
	printf("Caesar key is %li\n", k);

	int l = strlen(s);
	//int c;
	for (int c =0; c<l; c++)
	{	
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
