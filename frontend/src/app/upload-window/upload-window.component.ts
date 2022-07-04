import { Component, ElementRef, EventEmitter, HostListener, OnInit, Output, ViewChild } from '@angular/core';

@Component({
  selector: 'app-upload-window',
  templateUrl: './upload-window.component.html',
  styleUrls: ['./upload-window.component.css']
})
export class UploadWindowComponent implements OnInit {


  @ViewChild('uploadLink') uploadLink: ElementRef;
  @ViewChild('dragArea') dragArea: ElementRef;
  @Output() uploadImageEvent = new EventEmitter();
  quoteModalOpen: boolean;

  constructor() {
  }

  projectId: string;

  ngOnInit(): void {
  }

  @HostListener('dragover', ['$event'])
  public onDragOver(e: { preventDefault: () => void; stopPropagation: () => void; }): void {
      console.log(e);
      this.dragArea.nativeElement.style.visibility = 'visible';
      this.dragArea.nativeElement.style.opacity = 1;
      e.preventDefault();
      e.stopPropagation();
  }

  @HostListener('dragleave', ['$event'])
  public onDragLeave(e: { preventDefault: () => void; stopPropagation: () => void; }): void {
      this.dragArea.nativeElement.style.visibility = 'hidden';
      this.dragArea.nativeElement.style.opacity = 0;
      e.preventDefault();
      e.stopPropagation();
  }

  @HostListener('drop', ['$event'])
  public onDrop(e: { preventDefault: () => void; stopPropagation: () => void;
    dataTransfer: { files: any; }; target: { value: string; }; }): void {
      this.dragArea.nativeElement.style.visibility = 'hidden';
      this.dragArea.nativeElement.style.opacity = 0;
      e.preventDefault();
      e.stopPropagation();

      this.dispatchEvent(e.dataTransfer.files[0]);
      e.target.value = '';
  }

  onUploadAreaClicked(e: { preventDefault: () => void; stopPropagation: () => void; }): void {
      e.preventDefault();
      e.stopPropagation();
      this.uploadLink.nativeElement.click();
  }


  onFileSelected(e: { target: { files: any; value: string; }; }): void {
      this.dispatchEvent(e.target.files[0]);
      e.target.value = '';
  }

  dispatchEvent(file: any): void{
    this.uploadImageEvent.emit(file);
  }

}
