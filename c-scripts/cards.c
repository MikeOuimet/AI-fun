#include <stdio.h>

int main()

{
	int decks;
	puts("Enter the number of decks: ");
	scanf("%i", &decks);
	if (decks < 1) {
		puts("Not a valid number of decks");
		return 1;
	}
	printf("There are %i cards in the deck.\n", (decks*52));
	return 0;
}
