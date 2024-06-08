package filemanager

import (
	"fmt"
	"log/slog"
	"os"
)

var (
	fm *os.File
)

func Init(filename string) {
	var file *os.File
	file, err := os.Create(filename)
	if err != nil {
		slog.Error(fmt.Sprintf("error while creating file %v", filename))
		os.Exit(1)
	}

	fm = file
}

func Write(data any) {
	if fm == nil {
		slog.Error("empty file descriptor, strange")
		os.Exit(1)
	}

	fm.WriteString(fmt.Sprintf("%v", data))
}

func Close() {
	fm.Close()
}
