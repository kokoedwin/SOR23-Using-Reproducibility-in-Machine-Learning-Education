# source with wildcard
SOURCES := $(wildcard notebooks/*.md)
NBS := $(patsubst notebooks/%.md, notebooks/%.ipynb, $(SOURCES))

# rule to run
notebooks/%.ipynb: notebooks/%.md
	pandoc --embed-resources --standalone --wrap=none  $< -o $@
	pandoc --embed-resources --standalone --wrap=none  $(SOURCES) -o CutOut.ipynb

all: $(NBS) CutOut.ipynb

clean: 
	rm -f $(NBS) CutOut.ipynb

