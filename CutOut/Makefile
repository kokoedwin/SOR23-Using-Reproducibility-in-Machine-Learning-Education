SOURCES := CutOut.md
NBS := $(patsubst %.md,%.ipynb,$(SOURCES))

%.ipynb: %.md
	pandoc --embed-resources --standalone --wrap=none  $^ -o $@


all: $(NBS)

clean: 
	rm -f *.ipynb
