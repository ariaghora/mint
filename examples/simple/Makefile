CC ?= gcc

TARGET = simple
SRC = $(TARGET).c

$(TARGET).out: $(SRC)
	$(CC) $(CFLAGS) -o $@ $<

clean:
	rm -f $(TARGET).out
