import { HttpClient, HttpHeaders } from '@angular/common/http';
import { Component } from '@angular/core';
import { Observable } from 'rxjs';
import {delay } from 'rxjs/operators';

@Component({
  selector: 'app-root',
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.css']
})
export class AppComponent {

  private endpoint = 'http://localhost:8000/files/';
  public src: string;
  public file: File;
  selectedFile: any;
  isUploadWindowOpen = true;
  imagePath: any;
  url: any;
  detectedEmotion: any;
  isLoading = false;


  constructor(private http: HttpClient){
  }

  processImage(img: File): void{
    this.isLoading = true;
    this.showImage(img);
    this.selectedFile = img;
    const uploadData = new FormData();
    uploadData.append('myFile', this.selectedFile, this.selectedFile.name);
    this.uploadImage(uploadData)
    .pipe(delay(700))
    .subscribe((res) => {
      this.isLoading = false;
      this.isUploadWindowOpen = false;
      this.detectedEmotion = res;
      console.log(res);
    });
  }

    public uploadImage(uploadData: FormData): Observable<Response> {
      return this.http.post(this.endpoint, uploadData) as Observable<Response>;
    }

    showImage(file: File): void{
      const reader = new FileReader();
      this.imagePath = file;
      reader.readAsDataURL(file);
      reader.onload = (event) => {
          this.url = reader.result;
      };
    }

    onBackEvent(value: boolean): void{
      if (value){
        this.isUploadWindowOpen = true;
      }
    }

}
